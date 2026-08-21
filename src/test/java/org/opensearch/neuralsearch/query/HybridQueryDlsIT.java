/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.query;

import static org.junit.Assume.assumeTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.hc.core5.http.HttpHeaders;
import org.apache.hc.core5.http.ParseException;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.junit.After;
import org.junit.BeforeClass;
import org.opensearch.client.Request;
import org.opensearch.client.RequestOptions;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.XContentType;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.neuralsearch.OpenSearchSecureRestTestCase;

/**
 * Cross-plugin coverage for DLS applied to a Neural Search hybrid query.
 *
 * @see <a href="https://github.com/opensearch-project/neural-search/issues/1303">neural-search#1303</a>
 */
public class HybridQueryDlsIT extends OpenSearchSecureRestTestCase {

    private static final String INDEX_NAME = "hybrid-query-dls-test";
    private static final String SEARCH_PIPELINE_NAME = "hybrid-query-dls-test-pipeline";
    private static final String ROLE_NAME = "hybrid_query_dls_test_role";
    private static final String USER_NAME = "hybrid_query_dls_test_user";
    private static final String USER_PASSWORD = "HybridQueryDlsTest1!";
    private static final String DLS_QUERY = "{\"term\":{\"access\":\"allowed\"}}";
    private static final Set<String> ALL_DOCUMENT_IDS = Set.of("allowed-alpha", "allowed-beta", "blocked-alpha", "blocked-beta");
    private static final Set<String> ALLOWED_DOCUMENT_IDS = Set.of("allowed-alpha", "allowed-beta");

    @BeforeClass
    public static void requireSecurityPlugin() {
        assumeTrue("requires the Security plugin", Boolean.parseBoolean(System.getProperty("security.enabled", "false")));
    }

    @After
    public void cleanUpResources() throws IOException {
        deleteObjects(
            List.of(
                "/_plugins/_security/api/internalusers/" + USER_NAME,
                "/_plugins/_security/api/roles/" + ROLE_NAME,
                "/_search/pipeline/" + SEARCH_PIPELINE_NAME,
                "/" + INDEX_NAME
            )
        );
    }

    public void testDlsFiltersHybridHitsAndAggregations() throws Exception {
        createIndexAndDocuments();

        assertOk(performAdminJsonRequest("PUT", "/_search/pipeline/" + SEARCH_PIPELINE_NAME, """
            {
              "description": "Normalize hybrid query results for DLS testing",
              "phase_results_processors": [
                {
                  "normalization-processor": {
                    "normalization": { "technique": "min_max" },
                    "combination": { "technique": "arithmetic_mean" }
                  }
                }
              ]
            }
            """));
        assertCreatedOrOk(performAdminJsonRequest("PUT", "/_plugins/_security/api/roles/" + ROLE_NAME, roleBody()));
        assertCreatedOrOk(performAdminJsonRequest("PUT", "/_plugins/_security/api/internalusers/" + USER_NAME, userBody()));

        Map<String, Object> adminResponse = performSearch(primaryHybridRequest(), false);
        assertHitIds(adminResponse, ALL_DOCUMENT_IDS);
        assertGlobalAccessBuckets(adminResponse, Map.of("allowed", 2, "blocked", 2));

        Map<String, Object> restrictedResponse = performSearch(primaryHybridRequest(), true);
        assertHitIds(restrictedResponse, ALLOWED_DOCUMENT_IDS);
        assertGlobalAccessBuckets(restrictedResponse, Map.of("allowed", 2));

        Map<String, Object> adminResponseWithSuggest = performSearch(hybridRequestWithSuggest(), false);
        assertHitIds(adminResponseWithSuggest, ALL_DOCUMENT_IDS);
        assertSuggestionContains(adminResponseWithSuggest, "allowed-label-suggest", "document");

        Map<String, Object> restrictedResponseWithSuggest = performSearch(hybridRequestWithSuggest(), true);
        assertHitIds(restrictedResponseWithSuggest, ALLOWED_DOCUMENT_IDS);
        assertSuggestionContains(restrictedResponseWithSuggest, "allowed-label-suggest", "document");

        Map<String, Object> responseWithHybridFilter = performSearch(hybridRequestWithFilter(), true);
        assertHitIds(responseWithHybridFilter, Set.of("allowed-alpha"));

        Map<String, Object> singleClauseResponse = performSearch(singleClauseHybridRequest(), true);
        assertHitIds(singleClauseResponse, Set.of("allowed-alpha"));
    }

    private void deleteObjects(List<String> endpoints) throws IOException {
        IOException cleanupFailure = null;
        for (String endpoint : endpoints) {
            try {
                deleteObject(endpoint);
            } catch (IOException e) {
                if (cleanupFailure == null) {
                    cleanupFailure = e;
                } else {
                    cleanupFailure.addSuppressed(e);
                }
            }
        }
        if (cleanupFailure != null) {
            throw cleanupFailure;
        }
    }

    private void createIndexAndDocuments() throws IOException {
        Response createIndexResponse = performAdminJsonRequest("PUT", "/" + INDEX_NAME, """
            {
              "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
              },
              "mappings": {
                "properties": {
                  "access": { "type": "keyword" },
                  "signal": { "type": "keyword" },
                  "visibility": { "type": "keyword" },
                  "label": { "type": "text" }
                }
              }
            }
            """);
        assertEquals(RestStatus.OK.getStatus(), createIndexResponse.getStatusLine().getStatusCode());

        assertCreatedOrOk(performAdminJsonRequest("PUT", "/" + INDEX_NAME + "/_doc/allowed-alpha", """
            { "access": "allowed", "signal": "alpha", "visibility": "public", "label": "document" }
            """));
        assertCreatedOrOk(performAdminJsonRequest("PUT", "/" + INDEX_NAME + "/_doc/allowed-beta", """
            { "access": "allowed", "signal": "beta", "visibility": "private", "label": "manual" }
            """));
        assertCreatedOrOk(performAdminJsonRequest("PUT", "/" + INDEX_NAME + "/_doc/blocked-alpha", """
            { "access": "blocked", "signal": "alpha", "visibility": "public", "label": "confidential" }
            """));
        assertCreatedOrOk(performAdminJsonRequest("PUT", "/" + INDEX_NAME + "/_doc/blocked-beta", """
            { "access": "blocked", "signal": "beta", "visibility": "public", "label": "classified" }
            """));
        assertOk(client().performRequest(new Request("POST", "/" + INDEX_NAME + "/_refresh")));
    }

    private Map<String, Object> performSearch(String requestBody, boolean asDlsUser) throws IOException, ParseException {
        Request request = new Request("POST", "/" + INDEX_NAME + "/_search");
        request.addParameter("search_pipeline", SEARCH_PIPELINE_NAME);
        request.setJsonEntity(requestBody);
        if (asDlsUser) {
            RequestOptions.Builder options = RequestOptions.DEFAULT.toBuilder();
            String credentials = USER_NAME + ":" + USER_PASSWORD;
            options.addHeader(
                HttpHeaders.AUTHORIZATION,
                "Basic " + Base64.getEncoder().encodeToString(credentials.getBytes(StandardCharsets.UTF_8))
            );
            request.setOptions(options.build());
        }

        Response response = client().performRequest(request);
        assertEquals(RestStatus.OK.getStatus(), response.getStatusLine().getStatusCode());
        return XContentHelper.convertToMap(XContentType.JSON.xContent(), EntityUtils.toString(response.getEntity()), false);
    }

    private String primaryHybridRequest() {
        return """
            {
              "size": 10,
              "query": {
                "hybrid": {
                  "queries": [
                    { "term": { "signal": "alpha" } },
                    { "term": { "signal": "beta" } }
                  ]
                }
              },
              "aggs": {
                "all_documents": {
                  "global": {},
                  "aggs": {
                    "visible_access": {
                      "terms": { "field": "access" }
                    }
                  }
                }
              }
            }
            """;
    }

    private String hybridRequestWithSuggest() {
        return """
            {
              "size": 10,
              "query": {
                "hybrid": {
                  "queries": [
                    { "term": { "signal": "alpha" } },
                    { "term": { "signal": "beta" } }
                  ]
                }
              },
              "suggest": {
                "allowed-label-suggest": {
                  "text": "documnt",
                  "term": { "field": "label" }
                }
              }
            }
            """;
    }

    private String hybridRequestWithFilter() {
        return """
            {
              "size": 10,
              "query": {
                "hybrid": {
                  "queries": [
                    { "term": { "signal": "alpha" } },
                    { "term": { "signal": "beta" } }
                  ],
                  "filter": { "term": { "visibility": "public" } }
                }
              }
            }
            """;
    }

    private String singleClauseHybridRequest() {
        return """
            {
              "size": 10,
              "query": {
                "hybrid": {
                  "queries": [
                    { "term": { "signal": "alpha" } }
                  ]
                }
              }
            }
            """;
    }

    private void assertHitIds(Map<String, Object> responseBody, Set<String> expectedIds) {
        Map<String, Object> shards = asMap(responseBody.get("_shards"));
        assertEquals(0, ((Number) shards.get("failed")).intValue());

        Map<String, Object> hits = asMap(responseBody.get("hits"));
        Map<String, Object> total = asMap(hits.get("total"));
        assertEquals(expectedIds.size(), ((Number) total.get("value")).intValue());

        List<Map<String, Object>> hitList = asList(hits.get("hits"));
        List<String> hitIds = hitList.stream().map(hit -> (String) hit.get("_id")).collect(Collectors.toList());
        assertEquals("hybrid results must not contain duplicate documents", expectedIds.size(), hitIds.size());
        assertEquals(expectedIds, Set.copyOf(hitIds));
    }

    private void assertGlobalAccessBuckets(Map<String, Object> responseBody, Map<String, Integer> expectedBuckets) {
        Map<String, Object> aggregations = asMap(responseBody.get("aggregations"));
        Map<String, Object> allDocuments = asMap(aggregations.get("all_documents"));
        Map<String, Object> visibleAccess = asMap(allDocuments.get("visible_access"));
        List<Map<String, Object>> buckets = asList(visibleAccess.get("buckets"));
        Map<String, Integer> actualBuckets = buckets.stream()
            .collect(Collectors.toMap(bucket -> (String) bucket.get("key"), bucket -> ((Number) bucket.get("doc_count")).intValue()));

        assertEquals(
            expectedBuckets.values().stream().mapToInt(Integer::intValue).sum(),
            ((Number) allDocuments.get("doc_count")).intValue()
        );
        assertEquals(expectedBuckets, actualBuckets);
    }

    private void assertSuggestionContains(Map<String, Object> responseBody, String suggestionName, String expectedOption) {
        Map<String, Object> suggestions = asMap(responseBody.get("suggest"));
        List<Map<String, Object>> suggestionEntries = asList(suggestions.get(suggestionName));
        assertEquals(1, suggestionEntries.size());

        List<Map<String, Object>> options = asList(suggestionEntries.get(0).get("options"));
        Set<String> actualOptions = options.stream().map(option -> (String) option.get("text")).collect(Collectors.toSet());
        assertTrue(
            "expected suggestion options to contain " + expectedOption + " but got " + actualOptions,
            actualOptions.contains(expectedOption)
        );
    }

    private Response performAdminJsonRequest(String method, String endpoint, String body) throws IOException {
        Request request = new Request(method, endpoint);
        request.setJsonEntity(body);
        return client().performRequest(request);
    }

    private String roleBody() throws IOException {
        try (XContentBuilder builder = XContentFactory.jsonBuilder()) {
            builder.startObject()
                .field("cluster_permissions", List.of())
                .startArray("index_permissions")
                .startObject()
                .field("index_patterns", List.of(INDEX_NAME))
                .field("allowed_actions", List.of("read"))
                .field("dls", DLS_QUERY)
                .endObject()
                .endArray()
                .endObject();
            return builder.toString();
        }
    }

    private String userBody() throws IOException {
        try (XContentBuilder builder = XContentFactory.jsonBuilder()) {
            builder.startObject().field("password", USER_PASSWORD).field("opendistro_security_roles", List.of(ROLE_NAME)).endObject();
            return builder.toString();
        }
    }

    private void assertCreatedOrOk(Response response) {
        int status = response.getStatusLine().getStatusCode();
        assertTrue(
            "expected status 200 or 201 but got " + status,
            status == RestStatus.OK.getStatus() || status == RestStatus.CREATED.getStatus()
        );
    }

    private void assertOk(Response response) {
        assertEquals(RestStatus.OK.getStatus(), response.getStatusLine().getStatusCode());
    }

    private void deleteObject(String endpoint) throws IOException {
        try {
            Response response = client().performRequest(new Request("DELETE", endpoint));
            assertEquals(RestStatus.OK.getStatus(), response.getStatusLine().getStatusCode());
        } catch (ResponseException e) {
            if (e.getResponse().getStatusLine().getStatusCode() != RestStatus.NOT_FOUND.getStatus()) {
                throw e;
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object value) {
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Map<String, Object>> asList(Object value) {
        return (List<Map<String, Object>>) value;
    }
}
